@doc "

The univariate scalar a downstream observation observes for a composer.

An observation model observes one quantity, so lowering a composer first reduces
it to that quantity:

- a [`Convolved`](@ref) or [`Resolve`](@ref) is already univariate (the
  observed sum, resp. the marginal time-to-resolution) and is returned
  unchanged;
- a [`Sequential`](@ref) chain's observed quantity is the total elapsed time
  from origin to the terminal event, the convolution of its steps, returned as
  a [`Convolved`](@ref).

A [`Parallel`](@ref) has several independent endpoints and so no single observed
scalar; it is not lowered here.

# Examples
```@example
using ComposedDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
observed_distribution(seq)
```

# See also
- [`convolve_distributions`](@ref): the chain-step convolution
"
observed_distribution(d::UnivariateDistribution) = d

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
           convolve_distributions(leaves)
end

# A `Sequential` chain collapses to its observed convolved total, so
# `convolve_distributions` accepts the chain directly (the same collapse as
# `observed_distribution`, extending the ConvolvedDistributions verb to a
# composed stack).
function ConvolvedDistributions.convolve_distributions(d::Sequential)
    return observed_distribution(d)
end

# Flatten a composer's components to the univariate leaves whose sum is the
# chain's terminal time. A nested `Sequential` contributes its own steps; a
# nested `Parallel` has no single terminal time, so a chain step that is itself
# a `Parallel` cannot be collapsed and is rejected with a clear message.
function _observed_leaves(components::Tuple)
    leaves = UnivariateDistribution[]
    for c in components
        _append_observed_leaves!(leaves, c)
    end
    return leaves
end

_append_observed_leaves!(leaves, c::UnivariateDistribution) = push!(leaves, c)
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
