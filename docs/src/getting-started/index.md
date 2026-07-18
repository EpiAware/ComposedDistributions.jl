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

ComposedDistributions composes per-event delay distributions into one object that describes a whole record.
A composed object is a multi-state event process: named events linked by delays, which the composers wire into a tree.
The same object scores observed records with `logpdf` and simulates new ones with `rand`, so a model is built once and used in both directions.
It composes any [Distributions.jl](https://juliastats.org/Distributions.jl) `UnivariateDistribution`, with no censoring, so it is the generic composition layer.

The building blocks are five composers.
[`Sequential`](@ref) chains steps in series, [`Parallel`](@ref) fans branches off one shared origin, [`Resolve`](@ref) and [`Compete`](@ref) express one_of outcomes (a fixed-probability mixture and racing hazards), and [`Choose`](@ref) selects a branch from a data field.
The [`compose`](@ref) front-end lowers a NamedTuple, a Tables.jl table, or a nested matrix to the same stack.

The package has four layers, each building on the one before: leaves, composers, combination and lowering, and parameters and edits.
See the [Concepts](@ref concepts) page for the four layers in full and the verb that builds each one.

## A first example

A hospital pathway: an admission delay whose shape carries literature
uncertainty, then a death-versus-discharge split where the death probability
is the case-fatality ratio, alongside a separate delay to public reporting.

```@example overview
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
the `onset_admit` shape carries the uncertainty prior attached above, and the
death/discharge split shows up as its own `branch_probs` rows.

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

A literature-reported delay rarely comes with exact parameters. Wrap the
uncertainty inline with [`uncertain`](@ref): parameters that are themselves
distributions, nestable to any depth. The result is still a univariate
distribution, so it composes as a leaf everywhere (as `onset_admit` does
above), and `rand` draws the marginal (a fresh parameter draw each call); the
rest of the surface reports the template's central values until concrete
parameters are pinned with [`update`](@ref) (guard against a forgotten
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

## Getting help

For usage questions, ask on the [Julia Discourse](https://discourse.julialang.org)
(the SciML or usage categories) or the [epinowcast community forum](https://community.epinowcast.org),
our home for epidemiological modelling questions.
Please use [GitHub issues](https://github.com/EpiAware/ComposedDistributions.jl/issues)
for bug reports and feature requests only.
