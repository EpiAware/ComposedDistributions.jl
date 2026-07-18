# [Fitting a composed distribution](@id inference)

A composed tree carries its own estimation boundary.
The [`uncertain`](@ref) leaves mark which parameters are free, and everything else is held fixed.
A bare tree with no uncertain leaves estimates nothing, so fitting always starts by saying what is uncertain.

This page shows the three ways to fit a tree, all built on one PPL-neutral log-density.

## The log-density

[`as_logdensity`](@ref) packages a tree and its data into a log-density over just the estimated parameters.
The estimated parameters are the uncertain rows of [`params_table`](@ref), in that order, and [`flat_dimension`](@ref) counts them.

```@example inference
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]

prob = ComposedDistributions.as_logdensity(tree, data)
ComposedDistributions.flat_dimension(tree)
```

The one free parameter here is `onset_admit`'s shape.
[`logdensity`](@ref) scores a flat parameter vector, adding the priors' log-density to the data likelihood of the tree rebuilt at those values.

```@example inference
ComposedDistributions.logdensity(prob, [2.0])
```

Promote a fixed tree to estimate its free parameters with default priors through [`uncertain`](@ref)`(tree)` (equivalently `update(tree, param_priors(tree))`, the mechanism it is built on).

## Sampling without Turing

The assembled `prob` is a `LogDensityProblems` problem once the `LogDensityProblems` extension loads, so any consumer of that interface can sample it.
A gradient comes from wrapping it with `LogDensityProblemsAD` and a backend the codec differentiates under (ForwardDiff, ReverseDiff or Mooncake).

```julia
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff, AdvancedHMC

prob = as_logdensity(tree, data)
LogDensityProblems.dimension(prob)              # 1
grad = ADgradient(:ForwardDiff, prob)
# hand `grad` to AdvancedHMC / DynamicHMC / Pathfinder
```

Samplers work on an unconstrained vector, so a positive or simplex parameter needs its transform.
[`to_constrained`](@ref)`(prob, z)` returns the constrained parameters and the log-Jacobian for an unconstrained `z`, once `Bijectors` is loaded.

## Sampling with Turing

[`as_turing`](@ref) wraps the same log-density as a `DynamicPPL` model, so a tree is sampleable with Turing directly.
It is a light layer over the codec, with each estimated parameter a named site drawn from its own prior and the data likelihood added from the tree rebuilt at the draw.

```julia
using Turing

chain = sample(as_turing(tree, data), NUTS(), 1000)
```

## Reading the fit back

The site names match the readback, so a fitted chain reduces straight back onto the template.
[`chain_to_params`](@ref) reduces the draws to a nested parameter `NamedTuple`, and [`update`](@ref) rebuilds the tree at those values, collapsing every uncertain leaf.

```julia
using FlexiChains

fit = update(tree, chain)                        # the fitted tree
event(fit, :onset_admit)                         # a concrete Gamma
draws = param_draws(tree, chain)                 # every draw, for a posterior summary
```

## The tools

| Tool | What it gives | Loaded with |
|---|---|---|
| [`as_logdensity`](@ref) | the PPL-neutral log-density over the estimated parameters | base package |
| [`logdensity`](@ref) / [`flat_dimension`](@ref) | evaluate the density, count the parameters | base package |
| `LogDensityProblems` interface | `dimension` / `logdensity` / `capabilities` for any HMC consumer | `LogDensityProblems` |
| [`to_constrained`](@ref) | the unconstrained transform and its log-Jacobian | `Bijectors` |
| [`as_turing`](@ref) | a `DynamicPPL` model for `sample(...)` | `DynamicPPL` |
| [`chain_to_params`](@ref) / [`update`](@ref) | read a fitted chain back onto the tree | `DynamicPPL` and `FlexiChains` |
