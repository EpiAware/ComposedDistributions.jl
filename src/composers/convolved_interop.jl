# ============================================================================
# ConvolvedDistributions interop: composed trees feeding the convolution layer
# ============================================================================
#
# ComposedDistributions adds methods to ConvolvedDistributions' own generics so
# a composed tree feeds the convolution layer without the caller collapsing it
# by hand:
#   - `convolve_series(chain, series)`: the timeseries (renewal / latent)
#     convolution driven by a composed chain's observed total delay;
#   - `difference(chain, other)`: the difference of two observed totals;
#   - a `Convolved` / `Difference` node as a see-through composite leaf inside a
#     tree (its component parameters inventoried and fitted in place).
# Each collapses the composed operand to its observed univariate quantity
# (`observed_distribution`) and then reuses the univariate ConvolvedDistributions
# method unchanged. `difference` reuses it directly. `convolve_series` collapses
# to the observed total delay and hands it straight to
# `ConvolvedDistributions.convolve_series`: for a discrete delay that convolves
# directly; for a continuous delay (the common case — a chain's collapsed total
# is a `Convolved`) that throws upstream's own "discretise first" error rather
# than CD choosing a scheme on the caller's behalf (#226) — ConvolvedDistributions
# no longer discretises continuous delays itself (Convolved#68/#73); build the
# PMF with CensoredDistributions.jl (which owns primary and interval censoring)
# and convolve that.

# --- vector (timeseries) convolution driven by a composed chain --------------

@doc "

Convolve a timeseries through a composed chain's observed delay.

`convolve_series(chain, series)`, where `series` is a numeric timeseries
vector, collapses the [`Sequential`](@ref) chain to its observed total delay
([`observed_distribution`](@ref), the convolution of the chain steps) and
hands it straight to `ConvolvedDistributions.convolve_series`. With `series`
the expected events at unit-spaced times `0, 1, ..., t` (e.g. infections), a
discrete observed delay gives the expected downstream event counts at the
same times — the EpiNow2-style latent / renewal observation layer, driven by
a composed delay rather than a bare distribution.

A chain's observed total is usually continuous (e.g. a `Convolved` sum of
`Gamma`/`LogNormal` steps), and ConvolvedDistributions is discrete-convolution
-only: it throws, naming CensoredDistributions.jl (which owns primary and
interval censoring, including double-interval-censored masses for a
day-binned primary) as the way to build a PMF first, then
`convolve_series(pmf, series)`. This method does not choose a scheme on the
caller's behalf (#226) — it collapses the tree and delegates, nothing more.

Pass `events` to convolve the series to a chosen INTERIM event of the chain
rather than its endpoint. A single event name returns the count series at that
event; a tuple or vector of names returns a `NamedTuple` of series keyed by the
names. The cumulative delay to an interim event is the observed collapse of the
chain PREFIX up to that event (the convolution of the steps leading to it), so
selecting the terminal event reproduces the plain whole-chain result. Only a
plain continuous chain (every step a delay leaf, no branching) has such
per-event cumulative delays; a chain with a branching step is rejected.

# Arguments
- `chain`: a [`Sequential`](@ref) chain, collapsed to its observed total delay.
- `series`: the input timeseries (expected events at unit-spaced times from 0).

# Keyword Arguments
- `events`: a chain event name, or a tuple/vector of names, to convolve the
  series to (the cumulative delay of the chain prefix up to that event). The
  valid names are the chain's [`event_names`](@ref) after the origin. `nothing`
  (the default) convolves to the endpoint (the whole-chain observed total).

# Examples
```@example
using ComposedDistributions, ConvolvedDistributions, Distributions

chain = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
maxlag = length(infections) - 1
# Standing in for what CensoredDistributions.jl would build for the chain's
# own (continuous) observed total: a caller-owned PMF, here from a discrete
# distribution's own masses.
masses = pdf.(NegativeBinomial(5, 0.5), 0:maxlag)
pmf = ConvolvedDistributions.DelayPMF(masses, 1.0)
expected_counts = convolve_series(pmf, infections)

# The count series at named interim events (here the prefix to each event).
onset_to = sequential(:onset_admit => Gamma(2.0, 1.0),
    :admit_death => LogNormal(0.5, 0.4))
by_event = convolve_series(onset_to, infections; events = (:admit, :death))
```

# See also
- [`observed_distribution`](@ref): the chain-to-total-delay collapse.
- [`event_names`](@ref): the chain's event names, the valid `events` selectors.
- [`difference`](@ref): the difference of two observed totals.
"
function ConvolvedDistributions.convolve_series(
        d::Sequential, series::AbstractVector{<:Real}; events = nothing)
    events === nothing &&
        return convolve_series(observed_distribution(d), series)
    return _convolve_chain_events(d, series, events)
end

# A one_of node (`Resolve` / `Compete`) IS a univariate continuous delay — its
# marginal time to the resolving event — so `observed_distribution` returns it
# unchanged, and the collapse is handed to `ConvolvedDistributions.convolve_series`
# unchanged too — but `observed_distribution(d) === d` here (an AbstractOneOf
# is its own marginal), so a naive `convolve_series(observed_distribution(d),
# series)` re-dispatches to this same method and recurses forever. `invoke`
# forces dispatch as `ContinuousUnivariateDistribution` (the supertype
# `AbstractOneOf` claims, #29), landing on ConvolvedDistributions' own
# continuous-rejection method (discrete convolves; continuous throws
# upstream's "discretise first" error, same as the `Sequential` method above).
function ConvolvedDistributions.convolve_series(
        d::AbstractOneOf, series::AbstractVector{<:Real})
    return invoke(ConvolvedDistributions.convolve_series,
        Tuple{Distributions.ContinuousUnivariateDistribution,
            AbstractVector{<:Real}},
        d, series)
end

# A `Parallel` has several independent endpoints and so no single observed delay
# to convolve a series through; direct the caller to its branches. The `events`
# kwarg is accepted (and ignored) so passing it still lands on this informative
# error rather than a bare `MethodError`.
function ConvolvedDistributions.convolve_series(
        ::Parallel, ::AbstractVector{<:Real}; events = nothing)
    throw(ArgumentError(
        "cannot convolve a timeseries through a Parallel: it has several " *
        "independent observed endpoints and no single observed delay; " *
        "convolve each branch's chain separately, e.g. " *
        "`convolve_series(event(d, name), series)`"))
end

# A `Choose`'s observed delay depends on the data-selected alternative, so there
# is no single delay to convolve through; select an alternative first.
function ConvolvedDistributions.convolve_series(
        ::Choose, ::AbstractVector{<:Real}; events = nothing)
    throw(ArgumentError(
        "cannot convolve a timeseries through a Choose: its active " *
        "alternative is data-selected; convolve the chosen alternative, " *
        "e.g. `convolve_series(event(d, :index), series)`"))
end

# --- events-selected per-event convolution over a chain ----------------------
#
# `convolve_series(chain, series; events)` convolves the series to a named
# interim event of the chain: the cumulative delay to that event is the observed
# collapse of the chain PREFIX up to it (`_event_prefix_delay`), then the reused
# univariate series-through-a-delay method runs. A single name returns the series;
# a tuple/vector of names returns a `NamedTuple` of series keyed by the names.
# Only a plain continuous chain has per-event cumulative delays, so a branching
# step (whose flat events do not line up one-to-one with the delay steps) is
# rejected.

# A single event name: the series at that one event.
function _convolve_chain_events(
        d::Sequential, series::AbstractVector{<:Real}, name::Symbol)
    delay = _event_prefix_delay(d, name)
    return convolve_series(delay, series)
end

# Several event names: a `NamedTuple` of the per-event series, keyed by the names.
function _convolve_chain_events(
        d::Sequential, series::AbstractVector{<:Real}, names)
    syms = Tuple(names)
    all(n -> n isa Symbol, syms) || throw(ArgumentError(
        "convolve_series(..., events = ...): `events` must be an event " *
        "name or a tuple/vector of event names (Symbols); got $(typeof(names))"))
    series_by_event = map(n -> _convolve_chain_events(d, series, n), syms)
    return NamedTuple{syms}(series_by_event)
end

# The cumulative-delay distribution to a named interim event of a chain: the
# observed collapse of the prefix of delay steps leading to that event. The
# chain's flat `event_names` are `(origin, target_1, ..., target_k)`, one target
# per observed delay leaf, so the event at flat position `p` is reached by the
# first `p - 1` leaves; the origin (no elapsed delay) is not a convolvable event.
# The plain-chain guard rejects a branching step, whose flat events do not line
# up one-to-one with the observed delay leaves.
function _event_prefix_delay(d::Sequential, name::Symbol)
    has_uncertain(d) && throw(ArgumentError(
        "cannot select a cumulative-delay event on a chain with uncertain " *
        "leaves; pin the parameters with `update(tree, params)` first"))
    leaves = _observed_leaves(d.components)
    enames = event_names(d)
    length(enames) == length(leaves) + 1 || throw(ArgumentError(
        "convolve_series(..., events = $(repr(name))) needs a plain " *
        "continuous chain (every step a delay leaf, no branching); this " *
        "chain's events $(collect(enames)) do not line up one-to-one with " *
        "its delay steps"))
    idx = findfirst(==(name), enames)
    (idx === nothing || idx == 1) && throw(ArgumentError(
        "convolve_series(..., events = $(repr(name))): $(repr(name)) " *
        "is not a reachable interim event of this chain; valid events are " *
        "$(collect(enames[2:end])) (the origin $(repr(enames[1])) has no " *
        "elapsed delay to convolve)"))
    prefix = leaves[1:(idx - 1)]
    return length(prefix) == 1 ? only(prefix) : convolved(prefix)
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
forming the `Difference`. With both operands chains, `Z` is the difference
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
- `Difference`: the univariate difference distribution.
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

# --- Convolved / Difference as a see-through composite leaf ------------------
#
# A `Convolved` / `Difference` node used as a leaf is a pre-formed composite
# delay whose parameters ARE its components' parameters. The prior/params
# interface sees THROUGH it to the component leaves: `params_table` inventories
# each component's scalar parameters under a `component_i` path segment (so a
# two-Gamma `Convolved` at edge `:total` lists `total.component_1.shape`,
# `total.component_1.scale`, `total.component_2.shape`, ...), and `update`
# rebuilds the composite from the updated components. A component may itself be
# an [`uncertain`](@ref) / [`Varying`](@ref) leaf or a nested composite (the walk
# recurses), so the uncertain-first codec estimates a spec'd component parameter
# like any other leaf parameter.
#
# So the composite joins the shared `_node_children` / `_rebuild` walk the
# composer nodes use: its components ARE its node children, which lets the
# deferred-leaf guards (`has_uncertain` / `has_varying`) and `instantiate` recurse
# through it with the same machinery rather than a parallel per-composite path.
# It nonetheless stays a single flat SCORED slot (`length` 1) and an atomic node
# to the structural edits (`prune` / `splice` navigate by child name and do not
# read `_node_children`), so only the parameter inventory, reconstruction, and
# deferred-leaf resolution see inside it.

# A composite's node children are its component delays; `_rebuild` reassembles it
# from a new component tuple, preserving the solver method. Pairing these lets the
# composite ride every `_node_children` / `_rebuild` walk.
_node_children(d::Convolved) = d.components
_node_children(d::Difference) = (d.x, d.y)
_rebuild(d::Convolved, comps::Tuple) = Convolved(comps; method = d.method)
_rebuild(d::Difference, comps::Tuple) = Difference(comps[1], comps[2];
    method = d.method)

# The `component_i` path segment names for an `n`-component composite leaf,
# mirroring the edge/param naming the composers use for their named children.
_composite_component_names(n::Int) = ntuple(i -> Symbol(:component_, i), n)

# params_table rows: recurse into each component under a `component_i` segment,
# reusing the generic leaf walk for each component (so a plain, censored,
# uncertain, or nested-composite component is inventoried exactly as it would be
# as a standalone leaf, one row-group per component).
function _walk_rows!(edges, params_col, values, supports, priors, seen,
        d::Union{Convolved, Difference}, path)
    children = _node_children(d)
    names = _composite_component_names(length(children))
    for (name, child) in zip(names, children)
        _walk_rows!(edges, params_col, values, supports, priors, seen, child,
            (path..., name))
    end
    return nothing
end

# `update` rebuilds a composite from its updated components: the nested
# NamedTuple is keyed by `component_i`, one entry per component, each recursing
# through the leaf update (strict replace or merge-mode uncertainty). Mirrors the
# `Sequential`/`Parallel` recursion but rebuilds a `Convolved`/`Difference`,
# keeping the composite an atomic (single-slot) leaf everywhere else. A value
# aimed at the composite's own level (no `component_i` segment, whether a `Real`
# or a distribution) hits the shared unexpected-key check and errors informatively
# listing the component keys, rather than vanishing as a silent no-op.
function _update(d::Union{Convolved, Difference}, params::NamedTuple, shared,
        merge::Bool)
    children = _node_children(d)
    names = _composite_component_names(length(children))
    _check_child_keys(params, names, nameof(typeof(d)), shared)
    updated = ntuple(length(names)) do i
        _update(children[i], _child_params(params, names[i]), shared, merge)
    end
    return _rebuild(d, updated)
end

# Deferred-leaf resolution sees through a composite the same way, riding the
# shared `_node_children` / `_rebuild` walk: a composite reports `has_uncertain`
# / `has_varying` when any component does (so a fitting-loop guard and
# `observed_distribution`'s collapse guard catch an un-pinned / un-resolved
# component), and `instantiate` resolves each component against the context and
# reassembles the composite. These win over the univariate-leaf base cases
# (`false` / identity) because the concrete-type union is more specific.
has_uncertain(d::Union{Convolved, Difference}) = any(has_uncertain,
    _node_children(d))
has_varying(d::Union{Convolved, Difference}) = any(has_varying,
    _node_children(d))
function instantiate(d::Union{Convolved, Difference}, ctx::AbstractContext)
    return _rebuild(d, map(c -> instantiate(c, ctx), _node_children(d)))
end
