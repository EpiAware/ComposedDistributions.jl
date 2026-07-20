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

DistributionsInference.jl builds a `LogDensityProblems`-conformant problem for any composed tree, generically, over this same PPL-neutral core (via its fit-protocol extension, `parameter_rows`/`reconstruct`) — no weakdep extension in this package (#220).
A gradient comes from wrapping it with `LogDensityProblemsAD` and a backend the codec differentiates under (ForwardDiff, ReverseDiff or Mooncake).

```julia
using DistributionsInference, LogDensityProblems, LogDensityProblemsAD, ForwardDiff, AdvancedHMC

prob = DistributionsInference.as_logdensity(tree, data)
LogDensityProblems.dimension(prob)              # 1
grad = ADgradient(:ForwardDiff, prob)
# hand `grad` to AdvancedHMC / DynamicHMC / Pathfinder
```

Samplers work on an unconstrained vector, so a positive or simplex parameter needs its transform; DistributionsInference.jl's own `Bijectors` extension supplies it generically the same way.

## Sampling with Turing

DistributionsInference.jl's `as_turing` wraps the same log-density as a `DynamicPPL` model, so a tree is sampleable with Turing directly (#233 — this package no longer carries its own `as_turing`, which collided with this one when both packages were loaded).
It is a light layer over the codec, with each estimated parameter a named site drawn from its own prior and the data likelihood added from the tree rebuilt at the draw.

```julia
using DistributionsInference, Turing

chain = sample(DistributionsInference.as_turing(tree, data), NUTS(), 1000)
```

## Reading the fit back

Chain readback is DistributionsInference.jl's, not this package's own (#221): `DistributionsInference.readback` reduces a fitted chain straight to a rebuilt tree, collapsing every uncertain leaf; `readback_draws` keeps every draw for a posterior summary.
Both work generically over the fit-protocol core above, so the same two calls read back a tree fitted through `as_logdensity` or through `as_turing`.

```julia
using DistributionsInference, FlexiChains

fit = DistributionsInference.readback(tree, chain)      # the fitted tree
event(fit, :onset_admit)                                # a concrete Gamma
draws = DistributionsInference.readback_draws(tree, chain)   # every draw
```

## The tools

| Tool | What it gives | Loaded with |
|---|---|---|
| [`as_logdensity`](@ref) | the PPL-neutral log-density over the estimated parameters | base package |
| [`logdensity`](@ref) / [`flat_dimension`](@ref) | evaluate the density, count the parameters | base package |
| `DistributionsInference.as_logdensity` | the same core wrapped as a `LogDensityProblems` problem, generically | `DistributionsInference` |
| `DistributionsInference.to_constrained`-equivalent transform | the unconstrained transform and its log-Jacobian | `DistributionsInference` + `Bijectors` |
| `DistributionsInference.as_turing` | a `DynamicPPL` model for `sample(...)` | `DistributionsInference` + `DynamicPPL` |
| `DistributionsInference.readback` / `readback_draws` | read a fitted chain back onto the tree | `DistributionsInference` + `FlexiChains` |
