# [Getting started](@id getting-started)

Welcome to the `ComposedDistributions` documentation.
This page is the quickstart.
The home page is generated from the README, so it stays short; put the
walkthrough a new user needs here and grow it into tutorials as the package
develops.

## Installation

```julia
using Pkg
Pkg.add("ComposedDistributions")
```

Load the package:

```julia
using ComposedDistributions
```

## What ComposedDistributions does

A composed distribution is a multi-state event process: named events linked by delays, wired into a tree.
The same object scores an observed record with `logpdf` and simulates a new one with `rand`, so a model is built once and used in both directions.
See [Why ComposedDistributions?](../index.md) on the home page for the full motivation, and [Concepts](@ref concepts) for the four layers and the verb that builds each one.

## A first example

A hospital pathway: an admission delay with literature uncertainty on its
typical duration, then a death-versus-discharge split where the death
probability is the case-fatality ratio, alongside a reporting delay
truncated at a 21-day cutoff (reports arriving later are excluded) and a
referral delay censored at 14 days (a referral still pending at day 14 is
recorded as arriving then) — both plain Distributions.jl wrappers used here
as ordinary leaves.

```@example overview
using ComposedDistributions, Distributions, Random

cfr = 0.12   # case-fatality ratio among admitted cases

admission = @uncertain compose((
    path = sequential(
        :onset_admit => LogNormal(Normal(0.0, 0.2), 0.4),
        :admit_outcome => resolve(:death => (Gamma(1.5, 1.0), cfr),
            :discharge => Gamma(2.0, 1.5))),
    onset_report = truncated(Gamma(1.5, 1.0); upper = 21.0),
    onset_referral = censored(Gamma(1.0, 2.0); upper = 14.0)))
```

`admission` prints as the tree it is: three branches off the onset, the
admission branch itself a two-step chain ending in the death/discharge split,
and the reporting and referral branches keeping their `truncated()` and
`censored()` wrappers visible in the printed tree.

```@example overview
admission
```

The same object simulates a structured record and scores one straight back.

```@example overview
record = rand(Xoshiro(1), admission)
```

```@example overview
logpdf(admission, record)
```

Its free parameters read as a flat table, keyed by edge and parameter name;
`onset_admit`'s `mu` carries the uncertainty prior attached above (its
reported value, `0.0`, is the `LogNormal` family's own default, which here
also happens to be the prior's centre), and the death/discharge split shows
up as its own `branch_probs` rows.

```@example overview
params_table(admission)
```

[`event`](@ref) fetches any node by its dotted path, so the death outcome's
delay is reachable directly.

```@example overview
event(admission, :path, :admit_outcome, :death)
```

The tree still has an [`uncertain`](@ref) leaf, so it estimates rather than
scores at one fixed value until that leaf is pinned or fitted.

```@example overview
has_uncertain(admission)
```

Pinning `onset_admit` to a concrete delay collapses the admission chain to
one convolved total via [`observed_distribution`](@ref), integrating the
intermediate admission event out.

```@example overview
pinned = update(admission, (:path, :onset_admit) => LogNormal(1.5, 0.4))
total = observed_distribution(event(pinned, :path))
mean(total)
```

Fitting `admission` itself — estimating the uncertain leaf from data rather
than pinning it by hand — is one call away; see
[Fitting a composed distribution](@ref inference) for the full walkthrough.

## Uncertain distributions

A literature-reported delay rarely comes with exact parameters. Write the
uncertainty inline with [`@uncertain`](@ref) (as `onset_admit` does above): a
distribution literal in a parameter slot reads as that parameter's prior, the
natural spelling of the positional [`uncertain`](@ref) family form. The result
is still a univariate distribution, so it composes as a leaf everywhere, and
`rand` draws the marginal (a fresh parameter draw each call); the rest of the
surface reports a fixed placeholder for each uncertain parameter until
concrete values are pinned with [`update`](@ref) (guard against a forgotten
collapse with `has_uncertain`).

An [`uncertain`](@ref) leaf is one of two *deferred leaves*, resolved to a
concrete distribution later rather than fixed at build time.
See [Concepts](@ref concepts) for how it relates to its sibling
[`Varying`](@ref).

## Learning more

- Find the right verb by intent on the [Concepts](@ref concepts) page.
- Work through the composers end to end in [Composing distributions](@ref composing-distributions).
- See mutually exclusive outcomes in [Competing outcomes](@ref competing-outcomes) and multi-step delays in [Delay chains and the linear chain trick](@ref linear-chain).
- Want the full interface? See the [Public API](@ref public-api).
- Want the packages ComposedDistributions works alongside? See
  [Related packages](../index.md) on the home page.

## Getting help

For usage questions, ask on the [Julia Discourse](https://discourse.julialang.org)
(the SciML or usage categories) or the [epinowcast community forum](https://community.epinowcast.org),
our home for epidemiological modelling questions.
Please use [GitHub issues](https://github.com/EpiAware/ComposedDistributions.jl/issues)
for bug reports and feature requests only.
