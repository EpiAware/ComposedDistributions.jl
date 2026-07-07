# Inference-readback verbs: read a fitted Turing/FlexiChains chain back onto
# a composed-distribution template. `chain_to_params` and `param_draws` build
# the nested NamedTuple(s) [`update`](@ref) consumes; `strip_prefix` drops the
# outer submodel prefix from a chain's parameter names. All three are declared
# here as stubs (no method until both `DynamicPPL` and `FlexiChains` are
# loaded); the methods live in `ext/ComposedDistributionsFlexiChainsExt.jl`, so
# this package stays Turing-free until that extension is triggered.

@doc "

Read a fitted chain's parameters into the nested `NamedTuple` [`update`](@ref)
consumes.

`chain_to_params(template, chain)` walks `template` (a composed distribution,
matching the tree a `~ to_submodel(...)`-sampled parameters model was built
against) and reads each free parameter back from `chain` at its dotted
`prefix.edge.param` name, reducing multiple draws with `summary` (default
`mean`). Pair with [`update`](@ref):

```julia
ready = update(template, chain_to_params(template, chain))
```

or call `update(template, chain)` directly, which does the same in one step.

Reading a chain back onto a template that still holds an [`Uncertain`](@ref)
leaf collapses it to a concrete leaf at the read values (the same collapse
[`update`](@ref) always performs when given concrete parameters); a
[`Varying`](@ref) leaf keeps varying — only its `reference`'s fixed values are
updated, so [`has_varying`](@ref) is unchanged.

This method is available only when both `DynamicPPL` and `FlexiChains` are
loaded (the method lives in a package extension).

# Arguments
- `template`: the composed distribution the chain's parameters were sampled
  against.
- `chain`: the fitted `FlexiChains` chain to read parameter values from.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters were sampled under
  (default `:d`).
- `summary`: the reduction `AbstractVector -> scalar` applied to each
  parameter's draws (default `mean`).
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every draw.
- `draw`: a single iteration index to read (overrides `summary`/`draws`).

# Examples
```@example
using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
using FlexiChains: VNChain

template = compose((onset_admit = Gamma(2.0, 1.0),))

@model function onset_admit_model()
    shape ~ truncated(Normal(2.0, 0.3); lower = 0)
    scale ~ truncated(Normal(1.0, 0.3); lower = 0)
    return (shape = shape, scale = scale)
end
@model function tree_model()
    onset_admit ~ to_submodel(onset_admit_model())
    return (onset_admit = onset_admit,)
end
@model function fit()
    d ~ to_submodel(tree_model())
    return d
end

Random.seed!(1)
chain = sample(fit(), Prior(), 100; chain_type = VNChain, progress = false)
chain_to_params(template, chain)
```

# See also
- [`param_draws`](@ref): the vectorised, every-draw form.
- [`update`](@ref): the NamedTuple-keyed reconstruction this pairs with.
- [`strip_prefix`](@ref): drop the outer submodel prefix from a chain first.
"
function chain_to_params end

@doc "

Read every draw of a fitted chain into a vector of parameter `NamedTuple`s.

`param_draws(template, chain)` is the vectorised form of
[`chain_to_params`](@ref): where `chain_to_params` reduces the draws to one
`NamedTuple`, `param_draws` keeps every draw, so a per-draw distribution,
trajectory, or posterior-predictive summary maps `update` over the result:

```julia
draws = param_draws(template, chain)
update.(Ref(template), draws)
```

This method is available only when both `DynamicPPL` and `FlexiChains` are
loaded (the method lives in a package extension).

# Arguments
- `template`: the composed distribution the chain's parameters were sampled
  against.
- `chain`: the fitted `FlexiChains` chain to read every draw from.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters were sampled under
  (default `:d`).
- `draws`: a subset of iterations to keep (a range / index vector, or a
  predicate over the iteration index); `nothing` keeps every draw.

# Examples
```@example
using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
using FlexiChains: VNChain

template = compose((onset_admit = Gamma(2.0, 1.0),))

@model function onset_admit_model()
    shape ~ truncated(Normal(2.0, 0.3); lower = 0)
    scale ~ truncated(Normal(1.0, 0.3); lower = 0)
    return (shape = shape, scale = scale)
end
@model function tree_model()
    onset_admit ~ to_submodel(onset_admit_model())
    return (onset_admit = onset_admit,)
end
@model function fit()
    d ~ to_submodel(tree_model())
    return d
end

Random.seed!(1)
chain = sample(fit(), Prior(), 100; chain_type = VNChain, progress = false)
draws = param_draws(template, chain)
length(draws)
```

# See also
- [`chain_to_params`](@ref): the single-draw / reduced read this vectorises.
- [`update`](@ref): map over the result to rebuild a distribution per draw.
"
function param_draws end

@doc "

Drop the outer submodel prefix from a fitted chain's parameter names.

A parameter sampled through `d ~ to_submodel(...)` carries the submodel
prefix in its chain name (e.g. `d.onset_admit.shape`). `strip_prefix(chain)`
removes that one leading prefix, so [`chain_to_params`](@ref) /
[`param_draws`](@ref) can read it back with `prefix = Symbol(\"\")`.

This method is available only when both `DynamicPPL` and `FlexiChains` are
loaded (the method lives in a package extension).

# Arguments
- `chain`: the fitted `FlexiChains` chain to strip.

# Keyword Arguments
- `prefix`: the submodel variable name to remove (default `:d`).

# Examples
```@example
using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
using FlexiChains: FlexiChains, VNChain

@model function onset_admit_model()
    shape ~ truncated(Normal(2.0, 0.3); lower = 0)
    scale ~ truncated(Normal(1.0, 0.3); lower = 0)
    return (shape = shape, scale = scale)
end
@model function tree_model()
    onset_admit ~ to_submodel(onset_admit_model())
    return (onset_admit = onset_admit,)
end
@model function fit()
    d ~ to_submodel(tree_model())
    return d
end

Random.seed!(1)
chain = sample(fit(), Prior(), 100; chain_type = VNChain, progress = false)
stripped = strip_prefix(chain)
collect(FlexiChains.parameters(stripped))
```

# See also
- [`chain_to_params`](@ref), [`update`](@ref): read the (stripped) chain back.
"
function strip_prefix end
