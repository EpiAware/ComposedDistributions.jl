# [Frequently asked questions](@id faq)

Short answers to the questions this package tends to raise.
For a full walkthrough see [Composing distributions](@ref composing-distributions), and for the verb map see the [Concepts](@ref concepts) page.

## Where did censoring go?

ComposedDistributions is the generic composition layer, split out from [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl).
It composes any `Distributions.jl` `UnivariateDistribution` and adds no censoring of its own.
If you need primary-event censoring, interval censoring, or truncation of the observation process, use CensoredDistributions.jl, which builds its censored leaves on top of this composition algebra.
Right-truncation of a plain leaf is still available through `truncated()` from Distributions.jl, since any leaf is an ordinary distribution.

## Why does `rand` of a `Resolve` or `Compete` return a single number?

A one_of node is a univariate marginal: its `rand` is the time to resolution, whichever outcome occurs, and its `logpdf` scores that time.
To also learn which outcome occurred, use [`rand_outcome`](@ref), which returns an `(outcome, time)` pair.

```@example faq
using ComposedDistributions, Distributions, Random
import ComposedDistributions: rand_outcome

node = resolve(:death => (Gamma(1.5, 1.0), 0.3), :recover => Gamma(2.0, 1.5))
rand(Xoshiro(1), node), rand_outcome(Xoshiro(1), node)
```

The two share the same drawn time; `rand_outcome` just also names the outcome.

## How do I get the total-delay distribution of a chain?

A [`Sequential`](@ref) chain models each step separately, and its `rand` returns the per-step record.
To collapse the chain to the single distribution of its origin-to-final gap, use [`observed_distribution`](@ref), which convolves the steps into one delay.

```@example faq
chain = sequential(:onset_admit => Gamma(2.0, 1.0),
    :admit_death => Gamma(3.0, 1.0))
total = observed_distribution(chain)
mean(total)
```

For two standalone delays that are not part of a tree, [`convolved`](@ref) gives their sum `X + Y` directly.

## How do I fix a parameter across branches?

Use [`shared`](@ref) or [`tie`](@ref) to make several leaves one free parameter.
[`shared`](@ref) tags a leaf where it is built; [`tie`](@ref) walks an assembled tree to named leaves and ties them.
Either way [`params_table`](@ref) then inventories the tied occurrences once under the shared tag rather than as separate parameters.

```@example faq
d = compose((incubation = Gamma(2.0, 1.0),
    onset_report = Gamma(2.0, 1.0)))
tied = tie(d, :incubation, :onset_report; name = :delay)
unique(params_table(tied).edge)
```

## Why is my `logpdf` `-Inf`?

A composed `logpdf` is `-Inf` when a value falls outside a leaf's support.
The commonest cause is a negative or zero value scored against a positive-support delay, such as a `Gamma` or `LogNormal`.
When scoring a whole record, check that every value sits in its own leaf's support and that the record is laid out in the tree's event order (see [`event_names`](@ref)); a value in the wrong slot can land outside the leaf that scores it.

## How do I fit a composed distribution to data?

A composed object is a `Distribution`, so its `logpdf` is the likelihood term you need, and scoring is automatic-differentiation friendly.
Drop it into a [Turing.jl](https://github.com/TuringLang/Turing.jl) model, or any optimiser, exactly as you would any distribution; there is no package-specific fitting API.
[`params_table`](@ref) gives the free-parameter inventory and [`build_priors`](@ref) derives support-respecting priors to start from.

## Is a composed distribution really a `Distribution`?

Yes.
Every composer subtypes `Distributions.Distribution`, so `logpdf`, `rand`, `mean`, `var`, `cdf` (where defined) and the rest of the interface work unchanged, and a composed object drops into any code that expects a distribution.
