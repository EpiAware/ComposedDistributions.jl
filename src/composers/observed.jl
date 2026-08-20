@doc "

The univariate scalar a downstream observation observes for a composer.

An observation model observes one quantity, so lowering a composer first reduces
it to that quantity:

- a `Convolved` or [`Resolve`](@ref) is already univariate (the
  observed sum, resp. the marginal time-to-resolution) and is returned
  unchanged;
- a [`Sequential`](@ref) chain's observed quantity is the total elapsed time
  from origin to the terminal event, the convolution of its steps, returned as
  a `Convolved`.

A [`Parallel`](@ref) has several independent endpoints and so no single observed
scalar; it is not lowered here.

# Examples
```@example
using ComposedDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
observed_distribution(seq)
```

# See also
- `convolved`: the chain-step convolution
"
observed_distribution(d) = d

function observed_distribution(d::Sequential)
    # Structural errors (a Parallel step) first: they name the real
    # obstruction, which realising would not remove.
    leaves = _observed_leaves(d.components)
    # An uncertain leaf's template density/cdf is not the marginal, so the lazy
    # convolved total it would feed is not the observed quantity; fail here with
    # guidance rather than silently convolving the template values.
    has_uncertain(d) && throw(ArgumentError(
        "cannot collapse a chain with uncertain leaves to its observed " *
        "convolved total; pin the parameters with `update(tree, params)` to " *
        "collapse each uncertain leaf to its concrete template first"))
    return length(leaves) == 1 ? only(leaves) :
           convolved(leaves)
end

# A `Parallel` has several independent observed endpoints and so no single
# observed scalar to lower to; name the branches as the way forward.
function observed_distribution(::Parallel)
    throw(ArgumentError(
        "a Parallel has several independent observed endpoints and no single " *
        "observed scalar; lower each branch, e.g. " *
        "`observed_distribution(event(d, name))`"))
end

# A `Choose`'s observed quantity depends on the data-selected alternative, so it
# has no single lowering; direct the caller to the chosen alternative.
function observed_distribution(::Choose)
    throw(ArgumentError(
        "a Choose's observed quantity depends on the data-selected " *
        "alternative; lower the chosen alternative, e.g. " *
        "`observed_distribution(event(d, :index))`"))
end

# A `Sequential` chain collapses to its observed convolved total, so `convolved`
# accepts the chain directly (the same collapse as `observed_distribution`,
# extending the ConvolvedDistributions verb to a composed stack).
function ConvolvedDistributions.convolved(d::Sequential)
    return observed_distribution(d)
end

# A `Sequential` has no closed-form quantile of its own; its overall quantile
# is that of the collapsed observed total (a single leaf, or the `Convolved`
# sum `observed_distribution` builds). A one-step chain collapses to a bare
# leaf and needs nothing further; a multi-step chain needs `Convolved`'s own
# `quantile`, which only exists once `ConvolvedDistributionsOptimizationExt`
# is loaded (Optimization.jl + OptimizationOptimJL.jl) -- absent that, this
# throws the ordinary `MethodError` for the missing `Convolved` quantile.
quantile(d::Sequential, p::Real) = quantile(observed_distribution(d), p)

# Flatten a composer's components to the leaves whose sum is the chain's
# terminal time. A nested `Sequential` contributes its own steps; a nested
# `Parallel` has no single terminal time, so a chain step that is itself a
# `Parallel` cannot be collapsed and is rejected with a clear message. The
# accumulator is untyped so a duck-typed leaf (which need not subtype
# `UnivariateDistribution`) can be pushed, then `map(identity, ...)` narrows the
# element type on the way out: a chain of plain distributions must come back as
# a distribution vector to reach `convolved`'s
# `AbstractVector{<:UnivariateDistribution}` method.
function _observed_leaves(components::Tuple)
    leaves = Any[]
    for c in components
        _append_observed_leaves!(leaves, c)
    end
    return map(identity, leaves)
end

_append_observed_leaves!(leaves, c) = push!(leaves, c)
function _append_observed_leaves!(leaves, c::Sequential)
    for child in c.components
        _append_observed_leaves!(leaves, child)
    end
    return leaves
end
function _append_observed_leaves!(::Any, ::Parallel)
    throw(ArgumentError(
        "cannot collapse a Sequential chain whose step is a Parallel to a " *
        "single observed time; censor the Parallel's branches instead"))
end
