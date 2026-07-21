# Non-stationarity (a delay that changes with calendar time, a stratum, a
# region) is modelled here by generalising the leaf rather than adding a new
# composer verb; see the [`Varying`](@ref) and [`instantiate`](@ref)
# docstrings below.
#
# `Varying` is one of two DEFERRED LEAF types: a leaf that is not yet a concrete
# distribution but a map to one, delegating silently to a fallback until it is
# resolved, and guarded by a `has_*` predicate. The two cases differ only in
# what indexes the map and who fills the slot:
#
#   - `Varying` (here) maps an OBSERVED covariate (time, stratum) read from a
#     `Context`; resolved by `instantiate(tree, ctx)` looking the covariate up.
#   - `Uncertain` (`Uncertain.jl`) maps a LATENT parameter draw with a prior;
#     resolved by `rand` (the marginal) or collapsed by `update`.
#
# They share ONE resolution machinery: `instantiate` rebuilds through the same
# `_node_children` / `_rebuild` walk that `update`'s value collapse and
# `structural_edits.jl`'s path edits use, rather than hand-rolling its own tree
# walk.
# A leaf can be BOTH (a time-varying delay whose per-level parameter is itself
# `uncertain`): `instantiate` resolves the covariate and yields an `uncertain`
# leaf the estimation layer then reads as latent.

# --- the per-record / per-step context -------------------------------------

@doc "

Supertype of the covariate contexts a composed tree is resolved against.

A subtype carries the covariates a [`Varying`](@ref) leaf reads. [`Context`](@ref)
is the concrete open-`NamedTuple` implementation; the abstract type is what the
`uncertain distributions` work can extend with its own sampled-parameter
context. [`instantiate`](@ref) dispatches on `AbstractContext`.

# See also
- [`Context`](@ref): the concrete covariate bag.
- [`instantiate`](@ref): resolves a tree against a context.
"
abstract type AbstractContext end

@doc "

The covariate context a [`Varying`](@ref) leaf is resolved against.

A `Context` is an open bag of covariates (a `NamedTuple`) — calendar `time`, a
`region`/`stratum`, or (for the uncertain-distributions work) sampled parameter
values. [`instantiate`](@ref) reads the covariate a leaf names from it. Build one
with keyword covariates. It is open (rather than a fixed `time` field) so it can
also carry the uncertain-distributions work's sampled parameters.

# Examples
```@example
using ComposedDistributions

ctx = Context(time = 4.0)
ctx.covariates.time
```

# See also
- [`instantiate`](@ref): resolve a tree/leaf against a context.
- [`Varying`](@ref): the context-indexed leaf.
"
struct Context{NT <: NamedTuple} <: AbstractContext
    "The covariates keyed by name (`time`, `region`, sampled params, ...)."
    covariates::NT
end

Context(; covariates...) = Context(NamedTuple(covariates))

# Fetch a named covariate, with a clear error naming what was asked for and what
# the context actually carries (a missing covariate is a caller mistake, not a
# silent fallback).
function _covariate(ctx::Context, name::Symbol)
    haskey(ctx.covariates, name) || throw(ArgumentError(
        "context has no covariate $(repr(name)); it carries " *
        "$(collect(keys(ctx.covariates)))"))
    return ctx.covariates[name]
end

# Whether a context carries a named covariate. The `Choose`-on-the-seam path uses
# this to decide select-vs-resolve-all; a custom `AbstractContext` extends it.
_has_covariate(ctx::Context, name::Symbol) = haskey(ctx.covariates, name)

@doc "

Add or override covariates on a [`Context`](@ref), returning a new context.

`with_covariates(ctx; kwargs...)` is how the sampling layer of the *uncertain
distributions* work threads its LATENT index into the same covariate channel an
OBSERVED covariate uses: starting from a per-record observed context (calendar
`time`,
`region`), it adds the parameter values it has sampled
(`with_covariates(ctx; inc_shape = θ)`), and a [`Varying`](@ref) leaf keyed on
that name resolves against them exactly as a time-varying leaf resolves against
`time`. Observed and latent indices are the same covariate channel; the only
difference is who fills the slot — the data, or the sampler. Later keys win over
earlier ones.

# Arguments
- `ctx`: the [`Context`](@ref) to extend.

# Keyword Arguments
- `covariates...`: covariate `name = value` pairs to add or override.

# Examples
```@example
using ComposedDistributions

base = Context(time = 4.0)                       # observed covariates
drawn = with_covariates(base; inc_shape = 2.3)   # sampler adds a latent param
(drawn.covariates.time, drawn.covariates.inc_shape)
```

# See also
- [`Context`](@ref), [`instantiate`](@ref), [`Varying`](@ref).
"
function with_covariates(ctx::Context; covariates...)
    return Context(merge(ctx.covariates, NamedTuple(covariates)))
end

# --- the Varying leaf -------------------------------------------------------

@doc "

A context-indexed leaf: a delay whose distribution varies with a covariate.

`Varying` holds a map `f` from a covariate value to a `UnivariateDistribution`
(e.g. `t -> Gamma(shape(t), scale)`), the `covariate` name it reads from a
[`Context`](@ref) (default `:time`), and a `reference` distribution used whenever
the leaf is queried WITHOUT a context. Because `Varying <: UnivariateDistribution`
it drops into [`Sequential`](@ref) / [`Parallel`](@ref) / [`compose`](@ref) as an
ordinary leaf, and every `Distributions` method (`logpdf`, `cdf`, `mean`, `rand`,
`params`, ...) delegates to the `reference`, so a tree with varying leaves still
scores and samples at its reference by default.

[`instantiate`](@ref)`(d, ctx)` is the step that swaps the reference for `f`
evaluated at the context's covariate: `instantiate(leaf, Context(time = 4.0))`
returns `f(4.0)`. Resolve a whole tree at a context and then score / sample /
convolve the concrete result.

!!! warning
    Because the leaf delegates to `reference`, scoring or sampling a tree that
    still holds a `Varying` leaf does NOT error — it silently uses the reference
    (a wrong answer against real per-record covariates). Always
    [`instantiate`](@ref) first, and guard a fitting loop with
    [`has_varying`](@ref).

The varying map `f` is FIXED STRUCTURE (like a truncation bound or a censoring
window), so the introspection interface ([`params_table`](@ref), [`update`](@ref))
treats the `reference`'s parameters as the free parameters and peels/rewraps
through the wrapper; the coefficients of `f` are not (yet) inventoried (see the
design note's open questions).

# Fields
- `f`: map from a covariate value to a `UnivariateDistribution`.
- `covariate`: the [`Context`](@ref) field name to read (`Symbol`, default `:time`).
- `reference`: the distribution used when no context is supplied.

# See also
- [`varying`](@ref): friendly constructor.
- [`instantiate`](@ref): resolves the leaf against a context.
- [`Context`](@ref): the covariate bag.
"
struct Varying{F, D <: UnivariateDistribution} <: UnivariateDistribution{Continuous}
    "Map from a covariate value to a `UnivariateDistribution`."
    f::F
    "The `Context` field name this leaf reads (default `:time`)."
    covariate::Symbol
    "The distribution used when no context is supplied."
    reference::D
end

@doc "

Build a [`Varying`](@ref) context-indexed leaf.

`varying(f; covariate = :time, reference = f(0.0))` wraps a map `f` (a covariate
value to a `UnivariateDistribution`) as a leaf that varies with the named
`covariate`. The `reference` distribution is used when the leaf is queried
without a context; it defaults to `f(0.0)` (the map at the origin), which suits a
`:time` covariate — pass an explicit `reference` for a categorical covariate
where `f(0.0)` is not meaningful.

# Arguments
- `f`: map from a covariate value to a `UnivariateDistribution`.

# Keyword Arguments
- `covariate`: the [`Context`](@ref) field name to read (default `:time`).
- `reference`: the distribution used without a context (default `f(0.0)`).

# Examples
```@example
using ComposedDistributions, Distributions

# An onset->admit delay whose mean shortens over calendar time.
d = varying(t -> Gamma(2.0, 1.0 + 0.1 * t); covariate = :time)
mean(d)                                   # the reference (t = 0)
mean(instantiate(d, Context(time = 5.0))) # the delay at t = 5
```

# See also
- [`Varying`](@ref): the leaf type.
- [`instantiate`](@ref): resolve against a context.
"
function varying(f; covariate::Symbol = :time, reference = f(0.0))
    reference isa UnivariateDistribution || throw(ArgumentError(
        "varying `reference` must be a UnivariateDistribution; got " *
        "$(typeof(reference))"))
    return Varying(f, covariate, reference)
end

@doc "

Build a covariate-threshold [`Varying`](@ref) node: one subtree below a
covariate value, another at or above it.

`threshold(covariate, cutoff; below, above)` is the common activation shape
(a regime that switches at a given index value) as one call rather than a
hand-written `varying` map. Right-continuous at `cutoff`, matching the rest
of the ecosystem's step-function convention: `below` applies for a
covariate strictly less than `cutoff`, `above` for a covariate at or past
it. `below` and `above` may themselves be composite nodes (a `Resolve`/
`Compete`), since `threshold` builds on [`varying`](@ref).

# Arguments
- `covariate`: the [`Context`](@ref) field name to read.
- `cutoff`: the switching value; `above` applies at or past it.

# Keyword Arguments
- `below`: the subtree used for a covariate strictly less than `cutoff`.
- `above`: the subtree used for a covariate at or past `cutoff`.
- `reference`: the distribution used without a context (default `below`,
  the pre-threshold regime).

# Examples
```@example
using ComposedDistributions, Distributions

sw = threshold(:x, 10.0; below = Gamma(2.0, 1.0), above = Gamma(2.0, 3.0))
instantiate(sw, Context(x = 5.0))    # the `below` subtree
instantiate(sw, Context(x = 15.0))   # the `above` subtree
```

# See also
- [`varying`](@ref): the general covariate-indexed map this specialises.
"
function threshold(covariate::Symbol, cutoff::Real;
        below::UnivariateDistribution, above::UnivariateDistribution,
        reference::UnivariateDistribution = below)
    return varying(x -> x < cutoff ? below : above; covariate, reference)
end

# --- Distributions delegation (a Varying leaf behaves as its reference) -----
#
# Without a context a Varying leaf IS its reference distribution, so every
# scalar Distributions query delegates. This keeps a tree with varying leaves
# fully usable (score, sample, moments) at the reference, and is what makes the
# `instantiate` seam opt-in rather than mandatory.
params(d::Varying) = params(d.reference)
minimum(d::Varying) = minimum(d.reference)
maximum(d::Varying) = maximum(d.reference)
insupport(d::Varying, x::Real) = insupport(d.reference, x)
logpdf(d::Varying, x::Real) = logpdf(d.reference, x)
pdf(d::Varying, x::Real) = pdf(d.reference, x)
cdf(d::Varying, x::Real) = cdf(d.reference, x)
logcdf(d::Varying, x::Real) = logcdf(d.reference, x)
ccdf(d::Varying, x::Real) = ccdf(d.reference, x)
logccdf(d::Varying, x::Real) = logccdf(d.reference, x)
quantile(d::Varying, q::Real) = quantile(d.reference, q)
mean(d::Varying) = mean(d.reference)
var(d::Varying) = var(d.reference)
std(d::Varying) = std(d.reference)
sampler(d::Varying) = sampler(d.reference)
# `kwargs...` passes through a one_of reference's own `outcome`/`kind`
# keyword (e.g. `rand(varying_node; outcome = true)`), so a `Varying`
# wrapping a `Resolve`/`Compete` supports its reference's full sampling
# surface, not just the bare draw (#257).
Base.rand(rng::AbstractRNG, d::Varying; kwargs...) = rand(rng, d.reference; kwargs...)
function Base.rand(d::Varying; kwargs...)
    return rand(default_rng(), d.reference; kwargs...)
end
Base.rand(rng::AbstractRNG, d::Varying, n::Int) = rand(rng, d.reference, n)
Base.rand(d::Varying, n::Int) = rand(default_rng(), d.reference, n)
# A standalone one_of reference's own record-shaped `logpdf` (the shape its
# `rand` returns), so `logpdf(varying_node, rand(varying_node))` round-trips
# at the reference exactly as it does for a bare `Resolve`/`Compete` (#257).
logpdf(d::Varying, x::NamedTuple) = logpdf(d.reference, x)
probs(d::Varying) = probs(d.reference)

# The varying map is fixed structure; the reference carries the free parameters.
# Peel/rewrap through the wrapper so `params_table` / `update` see the inner free
# delay and a parameter update rebuilds the same varying leaf around it.
free_leaf(d::Varying) = free_leaf(d.reference)
function rewrap_leaf(d::Varying, inner)
    return Varying(d.f, d.covariate, rewrap_leaf(d.reference, inner))
end
_shared_tag(d::Varying) = _shared_tag(d.reference)
extra_leaf_params(d::Varying) = extra_leaf_params(d.reference)

function Base.show(io::IO, d::Varying)
    print(io, "Varying(", d.covariate, " -> ", d.reference, ")")
    return nothing
end

# --- node-aware delegation (#257) --------------------------------------------
#
# `Varying <: UnivariateDistribution` so a `reference` that is itself a
# composite univariate node (a `Resolve`/`Compete` one_of, the only
# univariate composers) is admitted by the `D <: UnivariateDistribution`
# bound already -- `varying(x -> resolve(...))` constructs today. What is
# missing is that every tree walker below sees `Varying` as an opaque leaf
# rather than looking through it at the reference's own node shape: the
# outcome-count, event-name and flat-slot machinery all special-case
# `AbstractOneOf` (and `Sequential`/`Parallel`/`Choose`, structurally
# unreachable here since they are multivariate) by CONCRETE type, so a
# `Varying` wrapping one falls to each function's generic leaf branch
# instead. Each delegation below forwards to `d.reference`, so a
# `Varying`-wrapped one_of node is stable and usable (event names, flat
# slot count, sampling, scoring) BEFORE any `instantiate` call, exactly as
# it is after one -- an un-resolved `Varying` behaves as its reference
# throughout, matching the rest of this file's delegation, not a special
# case for the composite reference. All are unconstrained on the reference
# type (not narrowed to `D <: AbstractOneOf`): for a plain leaf reference
# they resolve to the same generic leaf method as before, so this changes
# nothing there.

# The node-interface trio (`component_names`/`event_names`/`event_tree`):
# stable BEFORE resolution, reading the reference's own outcome/branch
# names rather than treating the wrapper as one flat leaf slot.
component_names(d::Varying) = component_names(d.reference)
event_names(d::Varying) = event_names(d.reference)
event_tree(d::Varying) = event_tree(d.reference)
# `event_tree`'s own composer methods recurse through this private
# NAME-carrying per-child helper (introspection.jl), not `event_tree`
# itself, so a `Varying` child needs its own entry here too: delegating to
# `_event_tree_child(name, d.reference)` (not the one-arg `event_tree`)
# keeps a plain-leaf reference on its existing `::Any` branch (the name,
# unchanged) and only a composite reference expands to its own nested
# NamedTuple.
_event_tree_child(name::Symbol, d::Varying) = _event_tree_child(name, d.reference)

# The flat, data-free value-vector contract (`child_nleaves`/`child_logpdf`/
# `child_rand!`, see nesting.jl): a `Varying` nested as a `Sequential`/
# `Parallel` child must occupy its reference's OWN slot width (a one_of
# reference's marginal time-to-resolution is one slot; a leaf reference is
# also one slot), not the generic-leaf default that happens to also be one
# slot for a leaf but silently mismatches sampling for a one_of reference
# (`rand` would try to write a one_of's full labelled RECORD into a single
# flat numeric slot without this, throwing).
child_nleaves(d::Varying) = child_nleaves(d.reference)
function child_logpdf(d::Varying, x, offset, n::Int)
    return child_logpdf(d.reference, x, offset, n)
end
# Disambiguates against the missing-aware `UnivariateDistribution` method
# above: without this, `x::AbstractVector{>:Missing}` matches both that
# method (via `Varying <: UnivariateDistribution`) and the generic one
# just above equally specifically. Delegates the same way.
function child_logpdf(
        d::Varying, x::AbstractVector{>:Missing}, offset, n::Int)
    return child_logpdf(d.reference, x, offset, n)
end
function child_rand!(out, offset, rng::AbstractRNG, d::Varying)
    return child_rand!(out, offset, rng, d.reference)
end

# The flat event-NAME tree walk (tree_events.jl): a `Varying` nested as a
# `Sequential`/`Parallel` child, or as a one_of outcome, contributes its
# reference's own event-name shape (one slot per one_of outcome) rather than
# a single positional slot.
_event_child_nleaves(d::Varying) = _event_child_nleaves(d.reference)
_one_of_outcome_slots(d::Varying) = _one_of_outcome_slots(d.reference)
_is_composer_outcome(d::Varying) = _is_composer_outcome(d.reference)
function _edge_origin_pair(edge_name::Symbol, child::Varying)
    return _edge_origin_pair(edge_name, child.reference)
end
function _walk_edge!(
        names, edge_name::Symbol, child::Varying, origin::Symbol, counter)
    return _walk_edge!(names, edge_name, child.reference, origin, counter)
end
function _walk_one_of_outcome!(
        names, oname::Symbol, delay::Varying, origin::Symbol, counter)
    return _walk_one_of_outcome!(names, oname, delay.reference, origin, counter)
end

# --- the resolution seam ----------------------------------------------------

@doc "

Resolve a composed tree (or a leaf) against a [`Context`](@ref).

`instantiate(d, ctx)` walks a composed distribution and returns the SAME tree
with every [`Varying`](@ref) leaf replaced by its distribution at the context's
covariate; a fixed leaf is returned unchanged (the identity default), so a
stationary tree is untouched and passing `nothing` is always a no-op. The result
is a fully concrete composer that scores, samples and convolves exactly as a
hand-built stationary tree would — non-stationarity is resolved HERE, before
those steps, so the convolution / renewal layer receives a concrete kernel per
context.

# Arguments
- `d`: a composer, a leaf, or a `Varying` leaf.
- `ctx`: a [`Context`](@ref) of covariates, or `nothing` (a no-op).

# Examples
```@example
using ComposedDistributions, Distributions

chain = sequential(:onset_admit => varying(t -> Gamma(2.0, 1.0 + 0.1t)),
    :admit_death => LogNormal(0.5, 0.4))
at_day5 = instantiate(chain, Context(time = 5.0))  # concrete, stationary chain
observed_distribution(at_day5)                     # the convolution kernel at t = 5
```

# See also
- [`Varying`](@ref), [`Context`](@ref): the leaf and the covariate bag.
- [`observed_distribution`](@ref): collapse the resolved chain to its kernel.
"
instantiate(d, ::Nothing) = d
instantiate(d::UnivariateDistribution, ::AbstractContext) = d
# Recurses into the produced subtree, not just `d.f(...)` alone: `f` can
# itself build a composite node (a `Resolve`/`Compete`) whose own outcomes
# may embed a further `Varying` leaf, and that nested leaf needs resolving
# against the SAME context too, so a single `instantiate` call fully
# resolves a `Varying` node regardless of how deep its produced subtree
# goes (#257). A no-op for the common case where `f` returns an
# already-stationary subtree.
function instantiate(d::Varying, ctx::AbstractContext)
    return instantiate(d.f(_covariate(ctx, d.covariate)), ctx)
end

# A composer resolves every child against the context and rebuilds itself
# unchanged, so the tree shape and names are preserved and only the leaves
# change. This reuses the `_node_children` / `_rebuild` reconstruction machinery
# that `update`'s value walk and `structural_edits.jl`'s path edits already
# share, so
# resolution is not a third hand-rolled tree walk. `Resolve` / `Compete` are
# `UnivariateDistribution`s, so these node methods win over the leaf identity.
function instantiate(d::Union{Sequential, Parallel, Resolve, Compete},
        ctx::AbstractContext)
    return _rebuild(d, map(c -> instantiate(c, ctx), _node_children(d)))
end

# A `Choose` selects an alternative by an OBSERVED data field (its `selector`).
# That is the categorical instance of covariate indexing, so it joins the same
# seam: if the context carries the selector covariate, `instantiate` SELECTS that
# alternative and resolves it (collapsing the disjunction to the chosen branch),
# unifying `Choose`'s `kind`-keyword dispatch with the continuous covariate case.
# Without the selector in the context there is no selection yet, so every
# alternative is resolved and the `Choose` is kept (the forward-simulation form).
function instantiate(d::Choose, ctx::AbstractContext)
    if _has_covariate(ctx, d.selector)
        return instantiate(_pick(d, _covariate(ctx, d.selector)), ctx)
    end
    return _rebuild(d, map(c -> instantiate(c, ctx), _node_children(d)))
end

# A tagged shared leaf keeps its tag through resolution (the resolved value is
# still the same shared parameter group); it is a wrapper leaf, so it forwards
# into its wrapped distribution rather than through `_node_children`.
function instantiate(d::Shared{tag}, ctx::AbstractContext) where {tag}
    return Shared{tag}(instantiate(d.dist, ctx))
end

# --- guarding against a forgotten `instantiate` -----------------------------

@doc "

Whether a composed distribution still contains an un-resolved [`Varying`](@ref) leaf.

A `Varying` leaf delegates every `Distributions` method to its `reference` until
the tree is resolved with [`instantiate`](@ref)`(tree, ctx)`, so scoring or
sampling a raw tree that still holds a `Varying` leaf SILENTLY uses the reference
(e.g. the `t = 0` delay) instead of the per-record value — a silent wrong answer,
not an error. Guard a scoring/sampling call in a fitting loop with this predicate:

```julia
resolved = instantiate(tree, Context(time = t))
@assert !has_varying(resolved)   # catch a forgotten instantiate before scoring
logpdf(resolved, x)
```

`has_varying` walks the tree and returns `true` as soon as any leaf is a
`Varying`; a fully stationary or fully-`instantiate`d tree returns `false`.

# Arguments
- `d`: the composed distribution, node, or leaf to check.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset = varying(t -> Gamma(2.0, 1.0 + 0.1t)),
    admit = LogNormal(0.5, 0.4)))
has_varying(tree)                                    # a varying leaf remains
has_varying(instantiate(tree, Context(time = 5.0)))  # resolved: false
```

# See also
- [`instantiate`](@ref): resolve every varying leaf against a context.
- [`Varying`](@ref): the context-indexed leaf.
- [`has_uncertain`](@ref): the same guard for the latent (uncertain) case.
"
has_varying(d::Varying) = true
has_varying(::UnivariateDistribution) = false
has_varying(d::Truncated) = has_varying(d.untruncated)
has_varying(d::Shared) = has_varying(d.dist)
# The composer nodes recurse through the shared `_node_children` accessor (the
# one `instantiate` also rebuilds through), so the guard is not a hand-rolled
# per-node walk; `has_uncertain` mirrors this.
function has_varying(d::Union{Sequential, Parallel, AbstractOneOf, Choose})
    return any(has_varying, _node_children(d))
end
