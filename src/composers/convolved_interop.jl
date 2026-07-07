# ============================================================================
# ConvolvedDistributions interop: composed trees feeding the convolution layer
# ============================================================================
#
# ComposedDistributions re-exports the ConvolvedDistributions verbs so a composed
# tree feeds the convolution layer without the caller collapsing it by hand:
#   - `convolve_distributions(chain, series)`: the timeseries (renewal / latent)
#     convolution driven by a composed chain's observed total delay;
#   - `difference(chain, other)`: the difference of two observed totals;
#   - a `Convolved` / `Difference` node as a fixed composite leaf inside a tree
#     (fixed structure for the flat params_table / update machinery).
# Each collapses the composed operand to its observed univariate quantity
# (`observed_distribution`) and then reuses the univariate ConvolvedDistributions
# method, so the composed and univariate results are identical by construction.

# --- vector (timeseries) convolution driven by a composed chain --------------

@doc "

Convolve a timeseries through a composed chain's observed delay.

`convolve_distributions(chain, series)`, where `series` is a numeric timeseries
vector, collapses the [`Sequential`](@ref) chain to its observed total delay
([`observed_distribution`](@ref), the convolution of the chain steps) and returns
the causal discrete convolution of `series` with that delay's discretised PMF,
truncated to the `series` window. With `series` the expected events at unit-spaced
times `0, 1, ..., t` (e.g. infections), the result is the expected downstream
event counts at the same times — the EpiNow2-style latent / renewal observation
layer, driven by a composed delay rather than a bare distribution.

The result is identical to
`convolve_distributions(observed_distribution(chain), series)`: the chain is
collapsed to its convolved total and then the univariate timeseries method runs.

# Arguments
- `chain`: a [`Sequential`](@ref) chain, collapsed to its observed total delay.
- `series`: the input timeseries (expected events at unit-spaced times from 0).

# Keyword Arguments
- `interval`: the discretisation grid width, which is also the series time-step.
  The series is unit-spaced, so this must be `1` (the default); any other value
  is rejected by the underlying univariate method.

# Examples
```@example
using ComposedDistributions, Distributions

chain = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
expected_counts = convolve_distributions(chain, infections)
```

# See also
- [`observed_distribution`](@ref): the chain-to-total-delay collapse.
- [`difference`](@ref): the difference of two observed totals.
"
function ConvolvedDistributions.convolve_distributions(
        d::Sequential, series::AbstractVector{<:Real}; interval = 1)
    return convolve_distributions(
        observed_distribution(d), series; interval = interval)
end

# A `Parallel` has several independent endpoints and so no single observed delay
# to convolve a series through; direct the caller to its branches.
function ConvolvedDistributions.convolve_distributions(
        ::Parallel, ::AbstractVector{<:Real}; interval = 1)
    throw(ArgumentError(
        "cannot convolve a timeseries through a Parallel: it has several " *
        "independent observed endpoints and no single observed delay; " *
        "convolve each branch's chain separately, e.g. " *
        "`convolve_distributions(event(d, name), series)`"))
end

# A `Choose`'s observed delay depends on the data-selected alternative, so there
# is no single delay to convolve through; select an alternative first.
function ConvolvedDistributions.convolve_distributions(
        ::Choose, ::AbstractVector{<:Real}; interval = 1)
    throw(ArgumentError(
        "cannot convolve a timeseries through a Choose: its active " *
        "alternative is data-selected; convolve the chosen alternative, " *
        "e.g. `convolve_distributions(event(d, :index), series)`"))
end

# --- difference of two observed totals ---------------------------------------

# Collapse a composed chain operand to its observed total delay; a bare
# distribution (including a univariate `Resolve` / `Compete` marginal) is already
# the observed quantity and passes through unchanged. After collapsing, both
# operands are univariate, so the univariate ConvolvedDistributions `difference`
# runs and there is no dispatch recursion (the collapsed operands are no longer
# `Sequential`).
_observed_operand(d::Sequential) = observed_distribution(d)
_observed_operand(d) = d

@doc "

Difference of two observed total delays, `Z = X - Y`.

`difference(a, b)` accepts a [`Sequential`](@ref) chain for either operand and
collapses it to its observed total delay ([`observed_distribution`](@ref)) before
forming the [`Difference`](@ref). With both operands chains, `Z` is the difference
of the two convolved totals; a bare distribution operand is used as-is. This
extends the univariate ConvolvedDistributions `difference` to composed stacks.

# Examples
```@example
using ComposedDistributions, Distributions

onset = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
report = Sequential(Gamma(1.5, 1.0), Gamma(1.0, 2.0))
gap = difference(onset, report)
```

# See also
- [`observed_distribution`](@ref): the chain-to-total-delay collapse.
- [`Difference`](@ref): the univariate difference distribution.
"
function ConvolvedDistributions.difference(a::Sequential, b; kwargs...)
    return difference(_observed_operand(a), _observed_operand(b); kwargs...)
end

function ConvolvedDistributions.difference(a, b::Sequential; kwargs...)
    return difference(_observed_operand(a), _observed_operand(b); kwargs...)
end

function ConvolvedDistributions.difference(
        a::Sequential, b::Sequential; kwargs...)
    return difference(_observed_operand(a), _observed_operand(b); kwargs...)
end

# --- Convolved / Difference as a fixed composite leaf ------------------------
#
# A `Convolved` / `Difference` node used as a leaf is a pre-formed composite
# delay: its parameters are its components' parameters (a nested tuple), not a
# flat scalar list. The flat params_table / update machinery keys leaves by a
# scalar parameter list, so it treats such a node as fixed structure — no
# free-parameter rows, and `update` leaves it unchanged (as with a `NoEvent`
# marker). To fit the components, compose them as explicit chain steps instead,
# so each component is its own inventoried leaf.

# No params_table rows: the composite leaf's parameters are fixed structure.
function _walk_rows!(edges, params_col, values, supports, priors, seen,
        ::Union{Convolved, Difference}, path)
    return nothing
end

# `update` leaves a fixed composite leaf unchanged (it emits no params_table rows
# to key an update against), in both strict and merge modes.
_update(d::Union{Convolved, Difference}, ::NamedTuple, shared, ::Bool) = d
