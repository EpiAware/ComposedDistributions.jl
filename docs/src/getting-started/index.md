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
    the package-owned source pages and let the scaffold render the rest.
    See [Infrastructure and template sync](@ref infrastructure) for how the
    kit keeps this repository in sync.

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

## Learning more

- Work through the composers end to end in [Composing distributions](@ref composing-distributions).
- Want the full interface? See the [Public API](@ref public-api).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ComposedDistributions.jl).
