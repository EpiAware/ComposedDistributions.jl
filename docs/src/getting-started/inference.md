# [Fitting a composed distribution](@id inference)

A composed tree carries its own estimation boundary.
The [`uncertain`](@ref) leaves mark which parameters are free, and everything else is held fixed.
A bare tree with no uncertain leaves estimates nothing, so fitting always starts by saying what is uncertain.

This page covers the codec this package owns, turning a tree's estimated parameters into a flat vector and back, and the log-density built on top.
For sampling and reading a fit back, see DistributionsInference.jl's inference guide.

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

## Sampling and reading the fit back

This package stops at the log-density.
It does not sample and it does not read a fit back onto the tree.
DistributionsInference.jl builds on the same codec (`flatten`/`unflatten`/`flat_dimension`/`params_table`) to wrap a tree as a `LogDensityProblems` problem or a `DynamicPPL` model, sample it with a gradient-based sampler or Turing, and read the fitted chain back onto the tree.
See DistributionsInference.jl's inference guide for the full walkthrough, including transforms for constrained parameters.

## The tools

| Tool | What it gives | Loaded with |
|---|---|---|
| [`as_logdensity`](@ref) | the PPL-neutral log-density over the estimated parameters | base package |
| [`logdensity`](@ref) / [`flat_dimension`](@ref) | evaluate the density, count the parameters | base package |
| [`flatten`](@ref) / [`unflatten`](@ref) | the flat-vector codec between a tree and its estimated parameters | base package |
| [`params_table`](@ref) | the estimated parameters as a `Tables.jl` table | base package |
