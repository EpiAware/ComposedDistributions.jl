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

## A first example

Compose delays into a chain with [`sequential`](@ref) (or build a whole tree
from a `NamedTuple` with [`compose`](@ref)). Edge names like `:onset_admit`
carry the event names, and a draw is labelled by them:

```@example quickstart
using ComposedDistributions, Distributions, Random

chain = sequential(:onset_admit => Gamma(2.0, 1.0),
    :admit_death => LogNormal(0.5, 0.4))
event_names(chain)
```

```@example quickstart
rand(Xoshiro(1), chain)
```

Overall moments collapse the chain to its observed total:

```@example quickstart
mean(chain)
```

Competing outcomes use [`resolve`](@ref) (fixed branch probabilities — the
death branch's probability is a CFR) or [`compete`](@ref) (racing hazards,
winning probabilities derived):

```@example quickstart
node = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => Gamma(2.0, 1.5))
winning_probabilities(node)
```

## Reading structure and building priors

[`params_table`](@ref) flattens a tree's free parameters into one row per
scalar, and [`build_priors`](@ref) turns it into a nested prior `NamedTuple`
(override only the rows you care about):

```@example quickstart
tbl = params_table(chain)
```

```@example quickstart
priors = build_priors(tbl)
priors.onset_admit
```

Fitted values go back in with [`update`](@ref), which returns a new tree of
the same shape:

```@example quickstart
update(chain, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)))
```

## Uncertain distributions

A literature-reported delay rarely comes with exact parameters. Wrap the
uncertainty inline with [`uncertain`](@ref): parameters that are themselves
distributions, nestable to any depth. The result is still a univariate
distribution, so it composes as a leaf everywhere, and `rand` draws the marginal
(a fresh parameter draw each call):

```@example quickstart
u = uncertain(Gamma, LogNormal(log(2.0), 0.2), 1.0)
rand(Xoshiro(2), u)
```

The positional family form above makes the shape uncertain and fixes the scale
at `1.0`; the keyword form
`uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))` is equivalent.
The rest of the univariate surface delegates to the template's central values,
so pin the parameters with [`update`](@ref) to collapse an uncertain leaf to a
concrete distribution on which the full ordinary surface works:

```@example quickstart
utree = sequential(:onset_admit => u, :admit_death => LogNormal(0.5, 0.4))
concrete = update(utree, (onset_admit = (shape = 2.0, scale = 1.0),
    admit_death = (mu = 0.5, sigma = 0.4)))
logpdf(concrete, [1.5, 0.8])
```

An uncertain parameter's spec also rides [`params_table`](@ref)'s `prior`
column, so [`build_priors`](@ref) picks it up without an explicit override:

```@example quickstart
build_priors(params_table(utree)).onset_admit.shape
```

## Learning more

- Want the full interface? See the [Public API](@ref public-api).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ComposedDistributions.jl).
