# Overall moments (mean/var/std) of a composed distribution are the standard
# `Distributions` moments on the composer itself: a scalar for a
# univariate-collapsible composer, a per-endpoint `Vector` for a `Parallel`.
# For a single event's own moment, use `event(d, name)`. See the docstrings
# below.

# --- per-leaf moment (free-delay transparent) ------------------------------

# Mean / variance of a (possibly censored) leaf: that of its inner free delay.
# A plain leaf is its own free leaf; a `Convolved` free-leafs to itself and so
# reuses its additive `mean`/`var`. An `Uncertain` leaf free-leafs straight to
# the template too (matching `mean`/`var(::Uncertain)`'s own delegation), so a
# tree containing one silently reports the template moment; guard with
# `has_uncertain` first if that matters.
_leaf_mean(leaf) = mean(free_leaf(leaf))
_leaf_var(leaf) = var(free_leaf(leaf))

# --- overall observed-level moments on the composer itself ------------------

@doc "

Overall mean of a composed distribution (the simple \"mean delay\").

`mean(d)` behaves like a normal delay distribution's mean. For a
univariate-collapsible composer (a [`Sequential`](@ref) chain, a
[`Convolved`](@ref), a [`Resolve`](@ref)) it returns the SCALAR mean of the
overall observed delay — the mean of [`observed_distribution`](@ref)`(d)` (the
convolved total for a chain, the marginal time-to-resolution for a `Resolve`).
For a genuinely multivariate [`Parallel`](@ref) (several independent observed
endpoints) it returns the per-ENDPOINT `Vector`, one overall mean per branch
endpoint, NOT the origin / intermediate events. Censoring is seen through to
the free delay. An [`uncertain`](@ref) leaf contributes its TEMPLATE moment
(parameter uncertainty is NOT propagated); guard with
[`has_uncertain`](@ref)`(d)` if that matters, draw the marginal with `rand`, or
collapse the leaf to its concrete template with [`update`](@ref)`(tree, params)`
to work with fixed parameters.

For a single event's own moment, fetch its distribution with [`event`](@ref)
and take its mean directly, e.g. `mean(event(d, :onset_admit))`.

# Examples
```@example
using ComposedDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
mean(seq)                 # overall mean delay (a scalar)
```

# See also
- [`var`](@ref), [`std`](@ref): the matching overall variance / std
- [`event`](@ref): fetch an edge/event's own distribution
- [`event_names`](@ref): the flat per-event labels
- [`observed_distribution`](@ref): collapse a chain to its terminal scalar
"
mean(d::Sequential) = _overall_moment(d, _leaf_mean)

@doc "

Overall variance of a composed distribution.

`var(d)` mirrors [`mean`](@ref): the scalar variance of the overall observed
delay for a univariate-collapsible composer (the variance of
[`observed_distribution`](@ref)`(d)`), or the per-ENDPOINT `Vector` for a
[`Parallel`](@ref). For a single event's own variance, use
`var(event(d, name))`.

# See also
- [`mean`](@ref), [`std`](@ref), [`event`](@ref)
"
var(d::Sequential) = _overall_moment(d, _leaf_var)

@doc "

Overall standard deviation of a composed distribution.

`std(d)` is `sqrt(var(d))` (or its elementwise form for a [`Parallel`](@ref)).
For a single event's own std, use `std(event(d, name))`.

# See also
- [`mean`](@ref), [`var`](@ref), [`event`](@ref)
"
std(d::Sequential) = sqrt(var(d))

# A `Parallel` is genuinely multivariate: its overall moment is the per-endpoint
# NamedTuple, one overall moment per branch endpoint keyed by `_endpoint_names`
# (a nested `Parallel` flattens its own endpoints in). The origin / intermediate
# events are not included; fetch a branch's own distribution via
# `event(d, name)` for its moment.
function mean(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, _leaf_mean))
end
function var(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, _leaf_var))
end
std(d::Parallel) = map(sqrt, var(d))

# `minimum`/`maximum` of a `Parallel` are per-endpoint too, mirroring `mean`:
# one support bound per branch endpoint keyed by `_endpoint_names`. A leaf branch
# contributes its own bound; a `Sequential` branch contributes the summed bound
# of its steps (the support of the convolved total); a nested `Parallel`
# flattens its endpoints in. Reusing `_endpoint_moment_vector` keeps the layout
# identical to `mean(d::Parallel)`.
function Base.minimum(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, minimum))
end
function Base.maximum(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, maximum))
end

# A generic `minimum`/`maximum` over any other composed node would otherwise
# fall through to the `Distributions` fallback, which tries to `iterate` the
# node and throws an opaque `MethodError`. Give the univariate-collapsible
# composers (a `Sequential` chain, a `Choose`) a clear error instead. The
# per-branch `Parallel` methods above and the marginal `Compete` / `Resolve`
# methods are more specific and still win.
function Base.minimum(d::AbstractComposedDistribution)
    throw(ArgumentError(
        "minimum(::$(nameof(typeof(d)))) is not defined for this composed " *
        "distribution; take it per-branch (`minimum(event(d, name))`), " *
        "per-step, or on the collapsed `observed_distribution(d)`"))
end
function Base.maximum(d::AbstractComposedDistribution)
    throw(ArgumentError(
        "maximum(::$(nameof(typeof(d)))) is not defined for this composed " *
        "distribution; take it per-branch (`maximum(event(d, name))`), " *
        "per-step, or on the collapsed `observed_distribution(d)`"))
end

# `Resolve` already defines the scalar univariate moment (the marginal
# time-to-resolution, `mean(as_mixture(c))`) in `Resolve.jl`, matching the
# overall semantics; no override needed here.

# A `Choose` has no single layout (the active alternative is data-selected), so a
# whole-tree moment is ill-defined; direct the caller to the chosen alternative.
function mean(::Choose)
    throw(ArgumentError(
        "mean(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `mean(event(d, :index))`"))
end
function var(::Choose)
    throw(ArgumentError(
        "var(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `var(event(d, :index))`"))
end
function std(::Choose)
    throw(ArgumentError(
        "std(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `std(event(d, :index))`"))
end

# --- the overall scalar moment of a univariate-collapsible node -------------
#
# `_overall_moment(d, f)` is the scalar overall moment of a node that collapses
# to a single observed univariate quantity. For independent steps the mean of the
# sum is the sum of means and the variance of the sum is the sum of variances, so
# a `Sequential` is the additive total over its free-delay leaf steps (the moment
# of `observed_distribution(d)`, computed free-delay-transparently). `f` is
# `_leaf_mean` or `_leaf_var`.
function _overall_moment(d::Sequential, f::F) where {F}
    sum(_overall_moment(c, f) for c in d.components)
end
_overall_moment(c::Resolve, ::typeof(_leaf_mean)) = _one_of_mix_mean(c)
_overall_moment(c::Resolve, ::typeof(_leaf_var)) = _one_of_mix_var(c)
# A racing-hazard node collapses to its marginal any-event (min) moment.
_overall_moment(c::Compete, ::typeof(_leaf_mean)) = mean(c)
_overall_moment(c::Compete, ::typeof(_leaf_var)) = var(c)
# A `Parallel` step inside a chain has several independent endpoints, so the chain
# has no single observed scalar to collapse to (mirroring `observed_distribution`,
# which rejects a `Sequential` whose step is a `Parallel`).
function _overall_moment(::Parallel, ::F) where {F}
    throw(ArgumentError(
        "cannot collapse a composer with a Parallel branch to a single overall " *
        "moment; take the moment of each `event(d, name)` branch instead"))
end
_overall_moment(leaf, f::F) where {F} = float(f(leaf))

# The per-endpoint moment Vector of a `Parallel`: one overall scalar moment per
# branch endpoint, in branch order. A nested `Parallel` branch contributes each
# of its own endpoints (flattened), so the vector length matches the number of
# independent observed endpoints. A `Sequential`/`Resolve`/leaf branch collapses
# to its single overall scalar via `_overall_moment`.
function _endpoint_moment_vector(d::Parallel, f::F) where {F}
    out = Float64[]
    for branch in d.components
        _append_endpoint_moments!(out, branch, f)
    end
    return out
end

function _append_endpoint_moments!(out, branch::Parallel, f::F) where {F}
    (for b in branch.components
            _append_endpoint_moments!(out, b, f)
        end; out)
end
function _append_endpoint_moments!(out, branch, f::F) where {F}
    push!(out, _overall_moment(branch, f))
end

# --- scalar marginal moment of an outcome / value child ---------------------

# The scalar marginal moment of an outcome / value child: a leaf's free-delay
# moment; a `Resolve`'s branch-prob-weighted mixture moment (over its free
# per-outcome moments, seeing through censored leaves, not the censored
# `mean`/`var(Resolve)`); a `Compete`'s marginal any-event moment.
_outcome_scalar_moment(leaf, ::typeof(_leaf_mean)) = _leaf_mean(leaf)
_outcome_scalar_moment(leaf, ::typeof(_leaf_var)) = _leaf_var(leaf)
function _outcome_scalar_moment(c::Resolve, ::typeof(_leaf_mean))
    return _one_of_mix_mean(c)
end
function _outcome_scalar_moment(c::Resolve, ::typeof(_leaf_var))
    return _one_of_mix_var(c)
end
# A racing-hazard outcome's scalar moment is the node's marginal any-event moment.
_outcome_scalar_moment(c::Compete, ::typeof(_leaf_mean)) = mean(c)
_outcome_scalar_moment(c::Compete, ::typeof(_leaf_var)) = var(c)

# A `Resolve`'s branch-prob-weighted mixture mean / variance, built from the
# free per-outcome moments so it sees through censored leaves (not the censored
# `mean(Resolve)`/`var(Resolve)`, which lower through `as_mixture` and have
# no analytic moment for a censored leaf).
function _one_of_mix_mean(c::Resolve)
    scalar_means = map(d -> _outcome_scalar_moment(d, _leaf_mean), c.delays)
    return sum(c.branch_probs .* scalar_means)
end

function _one_of_mix_var(c::Resolve)
    scalar_means = map(d -> _outcome_scalar_moment(d, _leaf_mean), c.delays)
    scalar_vars = map(d -> _outcome_scalar_moment(d, _leaf_var), c.delays)
    mix_mean = sum(c.branch_probs .* scalar_means)
    second = sum(c.branch_probs .* (scalar_vars .+ scalar_means .^ 2))
    return second - mix_mean^2
end
