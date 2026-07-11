# [Getting started](@id getting-started)

Welcome to the `ComposedDistributions` documentation.
This page is the quickstart.
The home page is generated from the README, so it stays short; put the
walkthrough a new user needs here and grow it into tutorials as the package
develops.

!!! note "These docs are generated"
    This site's layout, navigation, and infrastructure are produced by
    [EpiAwarePackageTools](https://github.com/EpiAware/EpiAwarePackageTools.jl).
    Editing the generated pages by hand is not needed; write your content in
    the package-owned source pages and let the scaffold render the rest. See
    the [EpiAwarePackageTools documentation](https://github.com/EpiAware/EpiAwarePackageTools.jl)
    for how the kit keeps this repository in sync.

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

The package has four layers, each building on the one before.

- **Leaves** are any Distributions.jl `UnivariateDistribution`, used directly as the per-event delays.
- **Composers** wire named leaves into an event tree.
- **Combination and lowering** join or collapse whole delays with [`convolved`](@ref), [`difference`](@ref) and [`observed_distribution`](@ref).
- **Parameters and edits** read and reshape an assembled tree with [`params_table`](@ref), [`build_priors`](@ref), [`update`](@ref), [`prune`](@ref) and [`splice`](@ref).

The [Concepts](@ref concepts) page maps each modelling concept to the verb that builds it.

## A first example

Compose two delays off a shared onset, then simulate and score a record.

```@example overview
using ComposedDistributions, Distributions, Random

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))

record = rand(Xoshiro(1), tree)
```

The composed object scores that record straight back.

```@example overview
logpdf(tree, record)
```

Read its free parameters as a flat table, keyed by edge and parameter name.

```@example overview
params_table(tree)
```

## Uncertain distributions

A literature-reported delay rarely comes with exact parameters. Wrap the
uncertainty inline with [`uncertain`](@ref): parameters that are themselves
distributions, nestable to any depth. The result is still a univariate
distribution, so it composes as a leaf everywhere, and `rand` draws the
marginal (a fresh parameter draw each call); the rest of the surface reports
the template's central values until you pin concrete parameters with
[`update`](@ref) (guard against a forgotten collapse with `has_uncertain`).

```@example overview
u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
utree = compose((onset_admit = u, admit_death = LogNormal(0.5, 0.4)))
has_uncertain(utree)
```

An [`uncertain`](@ref) leaf is one of two *deferred leaves*: a leaf that is not
yet a concrete distribution but a map to one, resolved before scoring. It maps a
**latent** parameter (a value a sampler draws, with the spec as its prior); its
sibling [`Varying`](@ref) maps an **observed** covariate (time, stratum),
resolved by [`instantiate`](@ref) against a [`Context`](@ref). Both delegate
silently to a fallback until resolved, and each has a guard —
`has_uncertain` / `has_varying` — for a fitting loop to check. See
[the varying-distributions reference](@ref varying-distributions) for the
observed case.

## Key features

- **Distributions.jl integration.** A composed object is a `Distribution`, so `logpdf`, `rand`, `mean`, `var` and the rest of the interface work unchanged, and any Distributions.jl leaf composes with no package-specific hooks.
- **One structure, many front-ends.** [`compose`](@ref) lowers a NamedTuple, a Tables.jl table, or a nested matrix to the same composer stack.
- **A readable, editable tree.** [`params_table`](@ref) inventories the free parameters, [`build_priors`](@ref) derives priors from their support, and [`update`](@ref) / [`prune`](@ref) / [`splice`](@ref) reshape the tree.
- **Convolution built in.** The package re-exports `ConvolvedDistributions`, so [`convolved`](@ref), [`difference`](@ref) and the quadrature surface are reachable through ComposedDistributions alone.
- **Automatic differentiation.** Scoring is differentiable through ForwardDiff, ReverseDiff, Mooncake and Enzyme, so a composed distribution drops into a probabilistic-programming fit.

## Learning more

- Find the right verb by intent on the [Concepts](@ref concepts) page.
- Work through the composers end to end in [Composing distributions](@ref composing-distributions).
- See mutually exclusive outcomes in [Competing outcomes](@ref competing-outcomes) and multi-step delays in [Delay chains and the linear chain trick](@ref linear-chain).
- Want the full interface? See the [Public API](@ref public-api).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ComposedDistributions.jl).
